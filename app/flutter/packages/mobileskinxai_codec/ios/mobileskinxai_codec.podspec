Pod::Spec.new do |s|
  s.name = 'mobileskinxai_codec'
  s.version = '1.0.0'
  s.summary = 'Verified fixed-policy native JPEG codec for MobileSkinXAI.'
  s.description = 'Build the native XCFramework using build_ios_codec.sh before pod install.'
  s.homepage = 'https://github.com/rinviriti/MobileSkinXAI'
  s.license = { :file => '../src/vendor/libjpeg-turbo/LICENSE.md' }
  s.author = { 'MobileSkinXAI' => 'research' }
  s.source = { :path => '.' }
  s.dependency 'Flutter'
  s.platform = :ios, '15.0'
  s.vendored_frameworks = 'Frameworks/MobileSkinXaiCodec.xcframework'
  s.user_target_xcconfig = { 'OTHER_LDFLAGS' => '$(inherited) -Wl,-u,_msx_decode -Wl,-u,_msx_roundtrip -Wl,-u,_msx_free -Wl,-u,_msx_error' }
end
